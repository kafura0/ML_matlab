function [ label_str ] = emolab2str( label )

switch label
    case 1
        label_str = 'anger';
    case 2
        label_str = 'disgust';
    case 3
        label_str = 'fear';
    case 4
        label_str = 'happiness';
    case 5
        label_str = 'sadness';
    case 6
        label_str = 'surprise';
    otherwise
        error('Invalid input');
end


end

