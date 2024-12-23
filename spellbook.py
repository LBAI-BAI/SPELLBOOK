import sys
import os
from lda import lda
from app import app
from hmp import hmp

def check_incantation(arg_list:list)->bool:
    """Check provided arguments, return False if spell is not known, inpurt file does not exist or number
    of provided arguments is not exactly equal to 3

    Args:
        - arg_list (list) : basically sys.argv

    Returns:
        - (bool) : True if all check passed, False if something is not right
    
    """

    # parameters
    check = True
    known_spell = ['lda', 'hmp']
    
    if len(arg_list) == 4:


        # check that spell is in spellbook
        if arg_list[1] not in known_spell:
            check = False

        # check that input file exist
        if not os.path.isfile(arg_list[2]):
            check = False

    else:
        check = False

    return check


def run(spell:str, input_file:str, output_folder:str):
    """Cast Spell !

    Args:
        - spell (str) : spell to cast
        - input_file (str) : path to input file
        - output_file (str) : path to output file
    
    """

    # deal with lda
    if spell == "lda":
        lda.run(input_file, output_folder)

    # deal with appariement
    if spell == 'app':
        app.run(input_file, output_folder)
        
    # deal with heatmap
    if spell == 'hmp':
        hmp.run(input_file, output_folder)

if __name__ == "__main__":


    if check_incantation(sys.argv):

        # parse incantation
        spell = sys.argv[1]
        input_file = sys.argv[2]
        output_folder = sys.argv[3]

        # init output
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        # cast spell
        run(spell, input_file, output_folder)

    else:
        print("[!] Incantation Failed !")

        

        
