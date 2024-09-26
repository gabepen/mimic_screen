import sqlite3

def multi_column_search(table_name: str, conditions: dict, db_path: str) -> list:

    """
    Perform a multi-column search on a SQLite database table. Thread safe version.
    Args:
        table_name (str): The name of the table to search.
        conditions (dict): A dictionary containing the search conditions. The keys represent the column names
            and the values represent the search values. For columns where multiple values can be accepted
            (e.g. list of values), use a list as the value. For columns where a single value is expected,
            use a string as the value.
        db_path (str): The file path to the SQLite database.
    Returns:
        list: A list of matching rows from the database table.
    Raises:
        None
    Example:
        conditions = {
            'column1': 'value1',
            'column2': ['value2', 'value3'],
            'column3': 'value4'
        }
        result = multi_column_search('table_name', conditions, '/path/to/database.db')
    """


    # open the connection here as concurrent use of the same connection object is not thread safe
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Generate the initial query string 
    query = f"SELECT * FROM {table_name} WHERE "
    query_parts = []
    query_values = []
    
    # generate the query parts and values
    for column, value in conditions.items():
        # for columns where multiple values can be accepted (e.g. list of values), use IN clause
        if isinstance(value, list):
            placeholders = ', '.join(['?'] * len(value))
            query_parts.append(f"{column} IN ({placeholders})")
            query_values.extend(value)
        # for columns where a single value is expected, use LIKE clause
        else:
            query_parts.append(f"{column} LIKE ?")
            query_values.append(f"%{value}%")
    
    # append these new clauses to the query
    query += ' AND '.join(query_parts)
    
    # execute the query
    cur.execute(query, query_values)
    
    # fetch the matching rows 
    rows = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return rows

def compare_rows(row1: list, row2: list) -> tuple:

    """
    Compare two rows and return a tuple containing a boolean value indicating whether the rows are equal and a list of indices where the rows differ.
    Args:
        row1 (list): The first row to compare.
        row2 (list): The second row to compare.
    Returns:
        tuple: A tuple containing a boolean value indicating whether the rows are equal and a list of indices where the rows differ.
    """
    
    if row1 == row2:
        return (True, [])
    else:
        diff = []
        for i, (item1, item2) in enumerate(zip(row1, row2)):
            if item1 != item2:
                diff.append(i)
        return (False, diff)
            
def main():
    
    # example usage
    conditions = {
        #'sourceTaxonIds': 'NCBI:txid246167',
        'sourceTaxonName': 'Vibrio crassostreae',
        'interactionTypeName': ['parasiteOf','hasHost','pathogenOf']
    }
    result = multi_column_search('interactions', conditions, '/path/to/globi.db')
        
if __name__ == '__main__':
    main()   

