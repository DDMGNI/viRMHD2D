

all:
	$(MAKE) -C common
	$(MAKE) -C linear
	$(MAKE) -C preconditioner
	$(MAKE) -C nonlinear
	

clean:
	$(MAKE) clean -C nonlinear
	$(MAKE) clean -C preconditioner
	$(MAKE) clean -C linear
	$(MAKE) clean -C common
	