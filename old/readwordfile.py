

def extract_text(word_file):
    """
	Extract text from .docx and .doc files, 
	 -> .docx extraction is the easy part
	 -> .doc extraction only work under windows machine

	Use the libraries:
			- platform
			- textract
			- win32com.client

	return the extracted text: a string if everything
	went right or the bool False if something went wrong.
    """

    ## importation
    import platform
    import textract
    
    ## init return value
    text = False

    ## detect the os
    opsystem = platform.system()

    ## Extract file extention
    extention = word_file.split(".")
    extention = extention[-1]

    ## Deal with docx (fast and easy !)
    if(extention == "docx"):

        ## extact text from word_file
        text = textract.process(word_file)


    ## When the fun begin ...
    elif(extention == "doc"):

    	## check if os is Windows
    	if(opsystem == 'Windows'):

    		## import specific windows lib
    		import win32com.client

    		## windows black magic
    		word = win32com.client.Dispatch("Word.Application")
    		word.visible = False
    		wb = word.Documents.Open(word_file)
    		doc = word.ActiveDocument

    		## extact text from word_file
    		try:
    			text = doc.Range().Text
    		except:
    			print("[!][ERROR] Something went wrong with the win32 API ...")
    			text = False

    		## close word application
    		word.Quit()

        ## doc file on non-Windows system
    	else:
    		print("[!][IN PROGRESS] Can't read .doc from "+str(opsystem)+" for now")


    ## return extracted text
    return(text)



def parse_text(text):
    """
    """

    ## importation
    import re

    ## preprocessing
    text = str(text)
    text = text.replace("\\xc3\\xa9", "é")

    ## parameters
    initial_patient = "NA"
    year_diag_sclerodermie_systemique = "NA"


    ## initial du patient
    match_inpat = re.findall('INITIALES DU PATIENT(.+)jamais rempli sur toutes les pages', text)
    if(match_inpat):
        inpat = match_inpat[0]
        inpat = inpat.replace('\\xc2\\xa0: ', '')
        inpat = inpat.replace('|', '')
        inpat = inpat.replace('_','')
        inpat = inpat.split('\\xd7\\x80')
        initial_patient = inpat[0]+"."+inpat[1]

    ## Year of diagnostic
    ## WARNING: mess up with chiffre order in the date
    ## TODO: Deal with non renseigne
    match_datediag = re.findall("Année du diagnostic de sclérodermie systémique(.{0,58})",text)
    if(match_datediag):
        date_diag = match_datediag[0]
        date_diag = date_diag.replace('\\n', '')
        date_diag = date_diag.replace('\\xc2\\xa0: ', '')
        date_diag = date_diag.split('|')
        date_diag = date_diag[1]
        date_diag = date_diag.replace("_", "")
        #date_diag = date_diag.split("\\xd7\\x80")
        date_diag = date_diag.replace("\\xd7\\x80", "")
        year_diag_sclerodermie_systemique = date_diag
        
        print(date_diag)


    #print(text)


def convert_file(data):
    data = str(data)
    data = data.replace("\\xc3\\xa9", "e")
    data = data.replace("\\xc2\\xab\\xc2\\xa0", "\"")
    data = data.replace("\\xc2\\xa0\\xc2\\xbb", "\"")
    data = data.replace("\\xc2\\xa0?", "?")
    data = data.replace("\\xe2\\x80\\x99", "\'")
    data = data.replace("\\xc3\\xa8", "e")
    data = data.replace("\\xc3\\xaa", "e")
    data = data.replace("\\n\\n", "")
    data = data.replace("\\xd7\\x80", "")
    data = data.replace("\\xef\\x82\\xa1", "")
    data = data.replace("\\xc2\\xa0", "")

    return str(data)



def parse_formated_text(text):
    """
    """

    print(text)


### TEST SPACE
test_file = "C:\\Users\\Immuno5\\Desktop\\Nathan\\SPELLBOOK\\Formular.docx"
test_file = "/home/nurtal/Spellcraft/MI/SPELLBOOK/Formular.docx"
stuff = extract_text(test_file)
text = convert_file(stuff)

#parse_text(stuff)
parse_formated_text(text)
