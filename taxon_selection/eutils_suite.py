from Bio import Entrez
Entrez.email = 'gapenunu@ucsc.edu'

class Librarian:

    ''' class for making entrez database operations on different 
        database types
    '''

    def __init__(self, db):

        self.db_type = db

    def update_db_type(self, new_db):
        
        self.db_type = new_db
    
    def find_ids_records(self, uids, retries=0):

        '''
        https://www.ncbi.nlm.nih.gov/books/NBK25498/#chapter3.EPost__ESummaryEFetch
        Uses epost -> esummary framework to pull records for list of UIDs
        '''

        # some times initial epost request fails retrying a couple times can 
        # yeild a succesful request.
        tries = 0
        while tries <= retries:
            request = Entrez.epost(self.db_type, id=','.join(uids))
            try:
                tries += 1
                result = Entrez.read(request)
            except RuntimeError as e:
                print('eutils ERROR:',e)
                print('retrying...')
                print(uids)

        webEnv = result["WebEnv"]
        queryKey = result["QueryKey"]
        data = Entrez.esummary(db=self.db_type, webenv=webEnv, query_key=queryKey)
        annotations = Entrez.read(data)

        #print("Retrieved %d annotations for %d genes" % (len(annotations), len(id_list)))
        return(annotations)

    def find_id_fasta(self, uid):

        ''' return a fasta file object for a uid entry 
        '''
        return Entrez.efetch(db=self.db_type, id=uid, rettype='fasta')
    
    def find_id_flatfile(self, uid):

        ''' returns a flatfile for a uid entry 
        '''
        return Entrez.efetch(db=self.db_type, id=uid, rettype='gb')

    def find_gid_from_taxid(self, taxid):

        ''' queries database with taxid
        '''
        try:  
            handle = Entrez.esearch(db=self.db_type, term="txid{}[orgn]".format(str(taxid)), idtype='gi') 
            record = Entrez.read(handle)
            handle.close()
            return record
        except RuntimeError as e:
                print('eutils ERROR:',e)
                print(taxid)

        
    
    def find_id_summary(self, uid):

        '''
        '''
        handle = Entrez.esummary(db=self.db_type, id=uid, report='full')
        record = Entrez.read(handle)
        handle.close()
        return record

def main():

    ''' testing here
    '''

    '''
    elib = Librarian('nuccore')
    recs = elib.find_ids_records(['CP092141.1', 'CP092139.1'])
    ff = elib.find_id_flatfile('CP092141.1')
    for l in ff:
        print(l)
        input()
    '''
    
    elib = Librarian('nuccore')
    prot_fa = elib.find_ids_records(['OW203764.1'])
    for l in prot_fa:
        print(l)
        input()

if __name__ == "__main__":
    main()