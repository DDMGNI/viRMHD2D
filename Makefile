
PYTHONPATH := $(CURDIR):${PYTHONPATH}
export PYTHONPATH


all:
	$(MAKE) -C rmhd
	

clean:
	$(MAKE) clean -C rmhd

