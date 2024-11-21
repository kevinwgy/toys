function parsave(fname, workspace)
    dirname  = [pwd '/mat_files/']
    if exist(dirname, 'dir')
        disp(['Writing data in ' dirname])
    else
        mkdir(dirname)
    end
    fname = [dirname fname '.mat'];
    save(fname, "-struct", "workspace", '-v7.3');
end
