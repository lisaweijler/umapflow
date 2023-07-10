

class TransformationConfig:

    def __init__(self, cutoff: bool,  
                 shuffle: bool, sequence_length: int, 
                 eventtype: str) -> None:

        if not isinstance(cutoff, bool):
            raise TypeError("cutoff must be a boolean")


        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")

        if not isinstance(sequence_length, int):
            raise TypeError("sequence_length must be an integer")
        
        eventtype_options = ['all', 'blast', 'nonblast', 'cd34total', 'bermude', 'bermude_cd34total']
        if not isinstance(eventtype, str) or eventtype not in eventtype_options:
            raise TypeError(f"eventtype must be a String and one of those options: {eventtype_options}")
       
        self.cutoff = cutoff
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.eventtype = eventtype
        
