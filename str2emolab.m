function [ label ] = str2emolab( label_str )

switch label_str
    case 'anger'
        label = 1;
    case 'disgust'
        label = 2;
    case 'fear'
        label = 3;
    case 'happiness'
        label = 4;
    case 'sadness'
        label = 5;
    case 'surprise'
        label = 6;
    otherwise
        error('Invalid input');
end

end

