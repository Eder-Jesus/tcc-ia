import pyodbc

db_config = {
    'url': '107.23.75.36,1433',
    'usuario': 'sa',
    'senha': 'Urubu100',
    'database': 'nutrilifelile',
}

connection_string = (
    f"DRIVER={{SQL Server}};"
    f"SERVER={db_config['url']};"
    f"DATABASE={db_config['database']};"
    f"UID={db_config['usuario']};"
    f"PWD={db_config['senha']};"
)

try:
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM Receita;")

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    conn.close()

except pyodbc.Error as ex:
    print(f"Erro na conexão com o banco de dados: {ex}")
