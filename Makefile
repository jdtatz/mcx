############################################################
#   MCX Monte Carlo Photon Simulator Packaging System
#
#  Author: Qianqian Fang <q.fang at neu.edu>
############################################################

PKGNAME=mcx
LIBNAME=libmcx.so
VERSION=2017.7
SOURCE=src
GUI=mcxstudio

all: bin/$(PKGNAME) bin/$(LIBNAME) bin/$(GUI) deb rpm

bin/$(PKGNAME):
	-$(MAKE) -C $(SOURCE) static
bin/$(LIBNAME):
	-$(MAKE) -C $(SOURCE) static
bin/$(GUI):
	-$(MAKE) -C $(GUI)
	-$(COPY) -a $(GUI)/debug/$(GUI) bin
deb: bin/$(PKGNAME)
	-package/mcxpkg/mcxdebmkdir.sh $(PKGNAME)
	-package/mcxpkg/mcxdebcopy.sh  $(PKGNAME) $(VERSION)
	-dpkg -b debian $(PKGNAME)-$(VERSION).deb
rpm:
	-package/mcxpkg/mcxrpmmkdir.sh $(PKGNAME)
	-package/mcxpkg/mcxrpmcopy.sh  $(PKGNAME) $(VERSION)
	cd rpmroot && tar zcvf $(PKGNAME)-$(VERSION).tar.gz $(PKGNAME) ; \
	rpmbuild --define="_topdir rpmroot/rpm" -ta $(PKGNAME)-$(VERSION).tar.gz
pymcx: bin/$(LIBNAME)
	-cp bin/$(LIBNAME) pymcx/$(LIBNAME)
	-python3 setup.py bdist_wheel
clean:
	-$(MAKE) -C $(SOURCE) clean
	-$(MAKE) -C $(GUI) clean
	-rm -rf debian rpmroot pkg.info $(PKGNAME)-$(VERSION).deb $(PKGNAME)-$(VERSION)*.rpm

.DEFAULT_GOAL := bin/$(PKGNAME)
