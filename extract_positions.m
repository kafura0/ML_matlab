function landmarks = extract_positions(file)


fid=fopen(file);
s=fscanf(fid,'%c');
landmarks=(str2num(s(findstr(s,'{')+1:findstr(s,'}')-1)))';
fclose(fid);

if size( landmarks, 1 ) > 66
    landmarks( [ 61,65 ], :, : ) = [];
end

end

