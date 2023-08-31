import os
import ast
from os import path
from configparser import ConfigParser, ExtendedInterpolation

class Empty:
    pass

class config_handler():
    
    def __init__(self, strConfigPath="config.ini"):
        
        strFilePath = path.dirname(path.abspath(__file__)) # current directory
        self.parser = ConfigParser(interpolation=ExtendedInterpolation())
        self.parser.read(os.path.join(strFilePath, strConfigPath))
        self.get_all_info()
      
    def get_all_info(self, ):
        
        print ("====== config info. ======")
        for strSection in self.parser.sections():
            for strOption in self.parser.options(strSection):
                print (f"  {strSection}: {strOption}:{self.parser.get(strSection, strOption)}")
        print ("==========================")
        
    def get_value(self, strSection, strOption, dtype="str"):
        
        if dtype == "str": return self.parser.get(strSection, strOption)
        elif dtype == "int": return self.parser.getint(strSection, strOption)
        elif dtype == "float": return self.parser.getfloat(strSection, strOption)
        elif dtype == "boolean": return self.parser.getboolean(strSection, strOption)
        elif dtype in {"list", "dict"}: return ast.literal_eval(self.parser.get(strSection, strOption))

    def set_value(self, strSection, strOption, newValue):
        
        if not self.parser.has_section(strSection): self.parser.add_section(strSection)
        self.parser[strSection][strOption] = str(newValue)

        if not hasattr(self, strSection): setattr(self, strSection, Empty())
        current_section = getattr(self, strSection)
        setattr(current_section, strOption, newValue)
        
    def member_section_check(self, strSection):
        return self.parser.has_section(strSection)
    
    def member_key_check(self, strSection, strOption):
        return self.parser.has_option(strSection, strOption)

if __name__ == "__main__":
    pass