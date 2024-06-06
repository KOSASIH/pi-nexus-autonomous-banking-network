import boto3

lake_formation = boto3.client('lakeformation')
glue = boto3.client('glue')

def create_data_lake(lake_name):
    # Create a new data lake using AWS Lake Formation
    response = lake_formation.create_data_lake(
        DataLakeName=lake_name,
        Admins=['arn:aws:iam::123456789012:role/data-lake-admin']
    )
    return response['DataLake']['DataLakeId']

def create_glue_catalog(catalog_name):
    # Create a new Glue catalog
    response = glue.create_catalog(
        CatalogName=catalog_name
    )
    return response['CatalogId']

def create_glue_table(table_name, database_name):
    # Create a new Glue table
    response = glue.create_table(
        DatabaseName=database_name,
        TableName=table_name,
        TableInput={
            'Name': table_name,
            'Description': 'Banking data table'
        }
    )
    return response['Table']['TableId']

if __name__ == '__main__':
    lake_name = 'banking-data-lake'
    catalog_name = 'banking-catalog'
    database_name = 'banking-database'
    table_name = 'banking-table'

    lake_id = create_data_lake(lake_name)
    catalog_id = create_glue_catalog(catalog_name)
    table_id = create_glue_table(table_name, database_name)
    print(f"Data lake created with ID: {lake_id}")
    print(f"Glue catalog created with ID: {catalog_id}")
    print(f"Glue table created with ID: {table_id}")
