#Written by Yukun Chen
#09/25/2017
project = cdiscount
home_ssh_location = yukun@68.232.116.191

HNAME=$(shell hostname)
ifeq ($(HNAME), wang-imac-01.ist.psu.edu)
	sshfile := ~/.ssh/id_imac_home
else ifeq ($(HNAME), yukun-mbp)
	sshfile := ~/.ssh/id_macbook_to_lionxv
endif

github_lionxv_credential:
	eval `ssh-agent -s`
	ssh-add ~/.ssh/id_github_from_cyberstar

# Use e.g. make ARGS="read_data/" home_upload_src
home_upload_src:
	@echo -n 'Copying to server...'
	ssh -p 5555 -i $(sshfile) $(home_ssh_location) "mkdir -p ~/work/$(project)/src/${ARGS}"
	scp -P 5555 -i $(sshfile) -r ./src/${ARGS}* $(home_ssh_location):~/work/$(project)/src/${ARGS}
	@echo ' done.'

# Use e.g. make ARGS="pert_mu2/hmms/" download_data
home_download_src:
	@echo -n 'Downloading from server...'
	@mkdir -p ./src/${ARGS}
	scp -P 5555 -i $(sshfile) -r $(home_ssh_location):~/work/$(project)/src/${ARGS}* ./src/${ARGS}
	@echo ' done.'

home_upload_data:
	@echo -n 'Copying to server...'
	ssh -p 5555 -i $(sshfile) $(home_ssh_location) "mkdir -p ~/work/$(project)/data/${ARGS}"
	scp -P 5555 -i $(sshfile) -r ./data/${ARGS}* $(home_ssh_location):~/work/$(project)/data/${ARGS}
	@echo ' done.'

# Use e.g. make ARGS="pert_mu2/hmms/" download_data
home_download_data:
	@echo -n 'Downloading from server...'
	scp -P 5555 -i $(sshfile) -r $(home_ssh_location):~/work/$(project)/data/${ARGS}* ./data/${ARGS}
	@echo ' done.'

testvars:
	@echo $(HNAME)
	@echo $(sshfile)
	@echo $(home_ssh_location)
