__author__ = 'kevintandean'
import requests, xmltodict
TOKEN = '9c4931c7-cea7-4e2e-af04-b5fd4f960e12'
#
# def get_smiles(id):
#     r = requests.get('http://www.chemspider.com/Search.asmx/GetCompoundInfo?CSID='+str(id)+'&token='+TOKEN)
#     try:
#         doc = xmltodict.parse(r.text)
#     except:
#         return None
#     try:
#         smiles = doc['CompoundInfo']['SMILES']
#     except KeyError:
#         smiles = None
#         pass
#     return smiles

def get_csid(name):
    url = 'http://www.chemspider.com/Search.asmx/SimpleSearch?query='+name+'&token='+TOKEN
    r = requests.get(url)
    try:
        doc = xmltodict.parse(r.text)
    except:
        return None
    try:
        csid = doc['ArrayOfInt']['int'][0]
        return csid
    except KeyError:
        pass

    return None

def get_compound_smiles_id(name):
    id = get_csid(name)
    smiles = get_smiles(id)
    return smiles

def get_smiles(name):
    url = 'http://cactus.nci.nih.gov/chemical/structure/'+name+'/smiles'
    r = requests.get(url)
    print r.text
    return r.text

get_smiles('thc')

