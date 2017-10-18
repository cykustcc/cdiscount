#Written by Yukun Chen
#09/25/2017
project = cdiscount
home_ssh_location = yukun@68.232.116.191

HNAME=$(shell hostname)
ifeq ($(HNAME), wang-imac-01.ist.psu.edu)
	sshfile := ~/.ssh/id_imac_home
	sshfile_ustc := ~/.ssh/id_imac_to_ustc
else ifeq ($(HNAME), yukun-mbp)
	sshfile := ~/.ssh/id_macbook_to_lionxv
	sshfile_ustc := ~/.ssh/id_imac_to_ustc
endif

github_lionxv_credential:
	eval `ssh-agent -s`
	ssh-add ~/.ssh/id_github_from_cyberstar

tensorflow:
	@echo -n "initiating tensorflow virtualenv..."
	./load_tensorflow.sh
	@echo 'done.'

# Use e.g. make ARGS="<file-name>.py" home_upload_src
ustc_upload_src:
	@echo -n 'Copying to server...'
	scp -i $(sshfile_ustc) -r ./src/${ARGS}* ustc:~/work/$(project)/src/${ARGS}
	@echo ' done.'

# Use e.g. make ARGS="<file-name>.py" ustc_download_data
ustc_download_src:
	@echo -n 'Downloading from server...'
	scp -i $(sshfile_ustc) -r ustc:~/work/$(project)/src/${ARGS}* ./src/${ARGS}
	@echo ' done.'

ustc_sync_pred:
	@echo -n 'Downloading from server...'
	rsync -v -r ustc:~/work/$(project)/data/pred/* ./data/pred/
	@echo ' done.'

ustc_sync_train_log:
	@echo -n 'Downloading from server...'
	rsync -v -r ustc:~/work/$(project)/train_log/* ./train_log/
	@echo ' done.'

# Use e.g. make ARGS="read_data/" home_upload_src
home_upload_src:
	@echo -n 'Copying to server...'
	scp -P 5555 -i $(sshfile) -r ./src/${ARGS}* $(home_ssh_location):~/work/$(project)/src/${ARGS}
	@echo ' done.'

# Use e.g. make ARGS="pert_mu2/hmms/" download_data
home_download_src:
	@echo -n 'Downloading from server...'
	scp -P 5555 -i $(sshfile) -r $(home_ssh_location):~/work/$(project)/src/${ARGS}* ./src/${ARGS}
	@echo ' done.'

home_upload_data:
	@echo -n 'Copying to server...'
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
