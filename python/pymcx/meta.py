# TODO Remove Python 3.6 support compat
# from __future__ import annotations
from abc import ABCMeta, abstractmethod
from importlib import import_module
from numbers import Number
from types import new_class
from typing import ClassVar, Dict, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
from typing_extensions import Annotated, Literal, get_args, get_origin

# This is a stable, public part of the cffi api
import _cffi_backend as ffi
from _cffi_backend import CType
try:
    # Unsure of the stability of this import
    from _cffi_backend import _CDataBase as CData
except:
    CData = TypeVar('CData')

# TODO Remove Python 3.6 support compat
try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = TypeVar("ArrayLike")
    DTypeLike = TypeVar("DTypeLike")


class StructField(metaclass=ABCMeta):
    name: str
    ctype: CType
    _ptr_field: str
    _owned_data_field: str

    def __init__(self, ctype, _ptr_field, _owned_data_field):
        self.ctype = ctype
        self._ptr_field = _ptr_field
        self._owned_data_field = _owned_data_field

    @abstractmethod
    def get(self, struct, instance) -> object:
        pass

    @abstractmethod
    def set(self, struct, instance, value):
        pass

    def owned_data(self, instance) -> dict:
        if not hasattr(instance, self._owned_data_field):
            setattr(instance, self._owned_data_field, dict())
        owned_data = getattr(instance, self._owned_data_field)
        return owned_data

    def __get__(self, instance, owner=None):
        struct = getattr(instance, self._ptr_field)
        return self.get(struct, instance)

    def __set__(self, instance, value):
        struct = getattr(instance, self._ptr_field)
        return self.set(struct, instance, value)

    def __set_name__(self, owner, name):
        self.name = name


T = TypeVar("T")


class GenericField(StructField, Generic[T]):
    _type: Literal[T]

    def __init__(
        self,
        _type: Literal[T],
        *args,
        **kwargs,
    ):
        self._type = _type
        super().__init__(*args, **kwargs)


S = TypeVar("S", bound=Number)


class ScalarField(GenericField[S]):
    def get(self, struct: CData, instance: object) -> S:
        return self._type(getattr(struct, self.name))

    def set(self, struct: CData, instance: object, value: S):
        setattr(struct, self.name, self._type(value))


NT = TypeVar("NT", bound=Tuple)


class NestedStructField(GenericField[NT]):
    def get(self, struct: CData, instance: object) -> NT:
        fields = self._type._fields
        nested = getattr(struct, self.name)
        return self._type(**{f: getattr(nested, f) for f in fields})

    def set(self, struct: CData, instance: object, value: NT):
        fields = self._type._fields
        tup = getattr(struct, self.name)
        for f, v in zip(fields, value):
            setattr(tup, f, v)


class _Array:
    def __getitem__(
        self,
        key: Union[Tuple[DTypeLike, int], Tuple[DTypeLike, int, Literal["C", "F"]]],
    ):
        dtype, ndim = key[:2]
        order = key[2] if len(key) > 2 else "C"
        _ = np.dtype(dtype)
        assert (
            isinstance(ndim, int) and ndim > 0
        ), "Use the Scalar dtype instead of a 0-dim Array"
        assert order == "C" or order == "F", "order must be either 'C' or 'F'"
        return Annotated[np.ndarray, dtype, ndim, order]


Array = _Array()


class ArrayField(StructField):
    def __init__(
        self,
        dtype: DTypeLike,
        ndim: int,
        order: Literal["C", "F"] = "C",
        *args,
        **kwargs,
    ):
        self.dtype = dtype
        self.ndim = ndim
        self.order = order
        self.ndptr = np.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, flags=order)
        super().__init__(*args, **kwargs)

    def get(self, struct: CData, instance: object) -> Optional[np.ndarray]:
        extra = self.owned_data(instance)
        return extra.get(self.name, None)

    def set(self, struct: CData, instance: object, value: ArrayLike):
        fs = np.dtype(self.dtype).fields
        if not isinstance(value, np.ndarray):
            value = np.asarray(list(value), dtype=self.dtype, order=self.order)
        elif (
            value.dtype != self.dtype
            and self.ndim == 1
            and value.ndim == 2
            and fs is not None
            and value.shape[1] == len(fs)
        ):
            # Added for compatibility, might deprecate or might keep it permanently
            value = np.asarray(
                list(map(tuple, value)), dtype=self.dtype, order=self.order
            )
        if fs is not None:
            value = value.view(np.recarray)
        _ptr = self.ndptr.from_param(value)
        setattr(struct, self.name, ffi.cast(self.ctype, value.ctypes.data))
        extra = self.owned_data(instance)
        extra[self.name] = value


def _origin_args(ty):
    origin = get_origin(ty)
    if origin is None:
        return ty, ()
    else:
        return origin, get_args(ty)


def _ty_to_field(ty: type, **kwargs) -> StructField:
    ty, args = _origin_args(ty)
    if issubclass(ty, Number):
        return ScalarField[ty](ty, **kwargs)
    elif hasattr(ty, "_fields"):
        return NestedStructField[ty](ty, **kwargs)
    elif ty is Annotated and args[0] is np.ndarray:
        assert len(args) == 4
        dtype, ndim, order = args[1:]
        return ArrayField(dtype, ndim, order, **kwargs)
    else:
        raise NotImplementedError(
            f"Type: {ty} with args: {args} to struct field creation is not yet implemented"
        )


class MetaStruct(type):
    def __new__(
        cls,
        clsname,
        bases,
        clsdict,
        *,
        ctype: CType,
        ptr_field: str = "_ptr",
        owned_data_field: str = "_owned_data",
        # cname_to_py: Optional[Dict[str, type]] = None,
    ):
        if not isinstance(ctype, CType):
            raise TypeError("`ctype` is not a CFFI CType")
        if not hasattr(ctype, "fields"):
            raise TypeError("`ctype` has no `fields`")
        fields = {f: v.type for f, v in ctype.fields}
        # __annotations__ can potentially be defined as None, not just undefined,
        # instead of a dict
        annotations = clsdict.get("__annotations__", None)
        if annotations is None:
            annotations = dict()
        eval_ctxt = vars(import_module(clsdict["__module__"]))
        annotations = {
            k: (eval(ty, eval_ctxt) if isinstance(ty, str) else ty)
            for k, ty in annotations.items()
        }
        for n, cty in fields.items():
            if n in annotations:
                if n in clsdict:
                    continue
                ty = annotations[n]
                clsdict[n] = _ty_to_field(
                    ty,
                    ctype=cty,
                    _ptr_field=ptr_field,
                    _owned_data_field=owned_data_field,
                )
            # elif cname_to_py is not None:
            #     # ty = cname_to_py[cty.cname]
            #     raise NotImplementedError(
            #         "Argument handling for `cname_to_py` is not yet implemented"
            #     )
            else:
                # Ignore or fill in with defaults ??
                pass
        return super().__new__(cls, clsname, bases, clsdict)


def field_annotation_docstrings(cls):
    import ast, inspect

    class Id(str):
        pass
    class Doc(str):
        pass

    class AnnotatedDocStringVisit(ast.NodeVisitor):
        def visit_AnnAssign(self, node):
            if node.simple:
                return Id(node.target.id)

        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return Doc(node.value) 

        def visit_Str(self, node):
            return Doc(node.s)

        def visit_Expr(self, node):
            return self.visit(node.value)

        def visit_ClassDef(self, node):
            yield from map(self.visit, ast.iter_child_nodes(node))

        def generic_visit(self, node):
            pass

    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        from itertools import tee
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    src = inspect.getsource(cls)
    parsed = ast.parse(src)
    cls_def, = parsed.body
    for i, d in pairwise(AnnotatedDocStringVisit().visit(cls_def)):
        if isinstance(i, Id) and isinstance(d, Doc):
            vars(cls)[str(i)].__doc__ = inspect.cleandoc(str(d))
    return cls
