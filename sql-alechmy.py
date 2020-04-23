#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import csv
import glob
import shutil
import psycopg2
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, MetaData, Table
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# PostgreSQL parameters
POSTGRES_ADDRESS = '12.64.211.212'
POSTGRES_PORT = '7001'
POSTGRES_USERNAME = 'ammar'
POSTGRES_PASSWORD = 'randomshit$'
POSTGRES_DBNAME = 'test'

# PostgreSQL connection string
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'
                .format(username=POSTGRES_USERNAME, 
                        password=POSTGRES_PASSWORD,
                        ipaddress=POSTGRES_ADDRESS,
                        port=POSTGRES_PORT,
                        dbname=POSTGRES_DBNAME))

# Create connection engine
conn = create_engine(postgres_str)
# #Then, print the names of the tables the engine contains using the .table_names()
print(conn.table_names())


# In[ ]:


## you could reflect an existing table like testing using the following code.

# Create a metadata object: metadata
metadata = MetaData()

# Reflect testing table from the engine: census
testing = Table('testing', metadata, autoload=True, autoload_with=conn)
print(repr(testing))


# In[ ]:


# Print the column names
print(testing.columns.keys())


# In[ ]:


# Print full metadata of testing
print(repr(metadata.tables['testing']))


# In[ ]:


print(testing.columns)


# In[ ]:


# Build select statement for census table: stmt
stmt = 'SELECT * FROM testing'

# Execute the statement and fetch the results: results
results = conn.execute(stmt).fetchall()

# Print results
print(results)


# In[ ]:


results[0]


# ## "Pythonic" way of interacting with databases 
#  When you used raw SQL in the last exercise, you queried the database directly. When using SQLAlchemy, you will go through a Table object instead, and SQLAlchemy will take case of translating your query to an appropriate SQL statement for you. So rather than dealing with the differences between specific dialects of traditional SQL such as MySQL or PostgreSQL, you can leverage the Pythonic framework of SQLAlchemy to streamline your workflow and more efficiently query your data. For this reason, it is worth learning even if you may already be familiar with traditional SQL. 
#  
#  In this exercise, you'll once again build a statement to query all records from the census table. This time, however, you'll make use of the select() function of the sqlalchemy module. This function requires a list of tables or columns as the only required argument: for example, select([my_table]). 
#  

# In[ ]:


# Import select
from sqlalchemy import select

# Reflect census table via engine: census
testing = Table('testing', metadata, autoload=True, autoload_with=conn)

# Build select statement for census table: stmt
stmt = select([testing])

# Print the emitted statement to see the SQL string
print(stmt)


# In[ ]:


# Execute the statement on connection and fetch 10 records: results
results = conn.execute(stmt).fetchall()
# Execute the statement and print the results
print(results)


# In[ ]:


# Execute the statement on connection and fetch 10 records: results
results = conn.execute(stmt).fetchmany(size=2)

# Execute the statement and print the results
print(results)


# In[ ]:


# Get the first row of the results by using an index: first_row
first_row = results[0]

# Print the first row of the results
print(first_row)


# In[ ]:


# Print the first column of the first row by accessing it by its index
print(first_row[0])

# Print the state column of the first row by using its name
print(first_row['name']," ... ", first_row['id'])


# In[ ]:


# Loop over the results and print the id , name
for result in results:
    print(result.id, result.name)


# In[ ]:


names = ['testing #1', 'testing #2']

#stmt = select([testing])
# stmt = stmt.where(testing.columns.name.in_(names))

stmt = select([testing]).where(testing.columns.name.in_(names))

# Loop over the ResultProxy and print id and name
for result in conn.execute(stmt):
    print(result.id, result.name)
    


# In[ ]:


# Import and_
from sqlalchemy import and_

stmt = select([testing])

stmt = stmt.where(
    # The state of California with a non-male sex
    and_(testing.columns.name == 'testing #2',
         testing.columns.id != "1"
         )
)

# Loop over the ResultProxy and print id and name
for result in conn.execute(stmt):
    print(result.id, result.name)
    


# In[ ]:


# Build a query to select the state column: stmt
from sqlalchemy import desc

stmt = select([testing.columns.name])

# Order stmt by the state column or multiple columns
stmt = stmt.order_by(desc(testing.columns.id))
stmt = stmt.order_by(testing.columns.id, desc(testing.columns.name))
# Execute the query and store the results: results
results = conn.execute(stmt).fetchall()

# Print the first 10 results
print(results[:10])


# In[ ]:


conn.table_names()


# In[ ]:


# # Import delete, select
# from sqlalchemy import delete, select

# # Build a statement to empty the census table: stmt
# delete_stmt = delete(data)

# # Execute the statement: results
# results = conn.execute(delete_stmt)

# # Print affected rowcount
# print(results.rowcount)

# # Build a statement to select all records from the census table : select_stmt
# select_stmt = select([data])

# # Print the results of executing the statement to verify there are no rows
# print(conn.execute(select_stmt).fetchall())


# In[ ]:


# # Build a statement to count records using the sex column for Men (M) age 36: count_stmt
# count_stmt = select([func.count(census.columns.sex)]).where(
#     and_(census.columns.sex == 'M',
#          census.columns.age == 36)
# )

# # Execute the select statement and use the scalar() fetch method to save the record count
# to_delete = connection.execute(count_stmt).scalar()

# # Build a statement to delete records from the census table: delete_stmt
# delete_stmt = delete(census)

# # Append a where clause to target Men ('M') age 36: delete_stmt
# delete_stmt = delete_stmt.where(
#     and_(census.columns.sex == 'M',
#          census.columns.age == 36)
# )


# In[ ]:


# Drop the state_fact tables
data.drop(conn)

# Check to see if state_fact exists
# Drop all tables
# metadata.drop_all(engine)

# Check to see if census exists
print(data.exists(conn))


# In[ ]:


# Import Table, Column, String, Integer, Float, Boolean from sqlalchemy
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255), unique=True),
             Column('count', Integer(), default=1),
             Column('amount', Float()),
             Column('valid', Boolean(), default=False)
)

# Use the metadata to create the table
metadata.create_all(conn)

# Print the table details
print(repr(metadata.tables['data']))


# In[ ]:


conn.table_names()


# In[ ]:


## inserting
# Import insert and select from sqlalchemy
from sqlalchemy import insert, select

# Build an insert statement to insert a record into the data table: insert_stmt
insert_stmt = insert(data).values(name='Anna', count=1, amount=1000.00, valid=True)

# Execute the insert statement via the connection: results
results = conn.execute(insert_stmt)

# Print result rowcount
print(results.rowcount)


# In[ ]:


# Build a select statement to validate the insert: select_stmt
select_stmt = select([data]).where(data.columns.name == 'Anna')

# Print the result of executing the query.
print(conn.execute(select_stmt).first())


# In[ ]:


# Build a list of dictionaries: values_list
values_list = [
    {'name': 'TerAnna', 'count': 1, 'amount': 1000.00, 'valid': True},
    {'name': 'Taylor', 'count': 1, 'amount': 750.00, 'valid': False}
]

# Build an insert statement for the data table: stmt
stmt = insert(data)

# Execute stmt with the values_list: results
results = conn.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)


# In[ ]:


# Build a select statement to validate the insert: select_stmt
select_stmt = select([data])

# Print the result of executing the query.
print(conn.execute(select_stmt).first())
print(conn.execute(select_stmt).fetchall())


#  using pandas. You can read a CSV file into a DataFrame using the read_csv() function (this function should be familiar to you, but you can run help(pd.read_csv) in the console to refresh your memory!). Then, you can call the .to_sql() method on the DataFrame to load it into a SQL table in a database. The columns of the DataFrame should match the columns of the SQL table.
# 
# .to_sql() has many parameters, but in this exercise we will use the following:
# 
#     name is the name of the SQL table (as a string).
#     con is the connection to the database that you will use to upload the data.
#     if_exists specifies how to behave if the table already exists in the database; possible values are "fail", "replace", and "append".
#     index (True or False) specifies whether to write the DataFrame's index as a column.
# 

# In[ ]:


# import pandas
import pandas as pd

# read census.csv into a dataframe : census_df
df = pd.read_csv("employee.csv")

# rename the columns of the census dataframe
df.columns = ['name', 'count', 'amount', 'valid']

# append the data from census_df to the "census" table via connection
df.to_sql(name="data", con=conn, if_exists="append", index=False)


# In[ ]:


df


# In[ ]:


select_stmt = select([data])

# Print the result of executing the query.
print(conn.execute(select_stmt).first())
print(conn.execute(select_stmt).fetchall())


# In[ ]:


from sqlalchemy import update

select_stmt = select([data]).where(data.columns.name == 'TerAnna')
results = conn.execute(select_stmt).fetchall()
print(results)
#print(results[0]['name'])

# Build a statement to update the fips_state to 36: update_stmt
update_stmt = update(data).values(name = "veranda")

# # Append a where clause to limit it to records for New York state
update_stmt = update_stmt.where(data.columns.name == 'Taylor')

# # Execute the update statement: update_results
update_results = conn.execute(update_stmt)



# In[ ]:


select_stmt = select([data])

# Print the result of executing the query.
print(conn.execute(select_stmt).first())
print(conn.execute(select_stmt).fetchall())


# In[ ]:


# Build a statement to update the notes to 'The Wild West': stmt
stmt = update(state_fact).values(notes='The Wild West')

# Append a where clause to match the West census region records: stmt_west
stmt_west = stmt.where(state_fact.columns.census_region_name == 'West')

# Execute the statement: results
results = connection.execute(stmt_west)

# Print rowcount
print(results.rowcount)


# In[ ]:


# Build a statement to select name from state_fact: fips_stmt
fips_stmt = select([state_fact.columns.name])

# Append a where clause to match the fips_state to flat_census fips_code: fips_stmt
fips_stmt = fips_stmt.where(
    state_fact.columns.fips_state == flat_census.columns.fips_code)

# Build an update statement to set the name to fips_stmt_where: update_stmt
update_stmt = update(flat_census).values(state_name=fips_stmt)

# Execute update_stmt: results
results = connection.execute(update_stmt)

# Print rowcount
print(results.rowcount)


# In[ ]:


from sqlalchemy import func
# Build a query to count the distinct states values: stmt
stmt = select([func.count(data.columns.name.distinct())])

# Execute the query and store the scalar result: distinct_state_count
distinct_state_count = conn.execute(stmt).scalar()

# Print the distinct_state_count
print(distinct_state_count)


# In[ ]:


select_stmt = select([data])
# Print the result of executing the query.
print(conn.execute(select_stmt).fetchall())
print(len(conn.execute(select_stmt).fetchall()))


# In[ ]:


from sqlalchemy import func

#    {'name': 'TerAnna', 'count': 1, 'amount': 1000.00, 'valid': True},
# Build a query to select the state and count of ages by state: stmt
stmt = select([data.columns.name, func.count(data.columns.name)])

# Group stmt by state
stmt = stmt.group_by(data.columns.name)

# Execute the statement and store all the records: results
results = conn.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())


# In[ ]:


# Import func
from sqlalchemy import func

# Build an expression to calculate the sum of pop2008 labeled as population
salary_sum = func.sum(data.columns.amount).label('aggregate salary')

# Build a query to select the state and sum of pop2008: stmt
stmt = select([data.columns.name, salary_sum])

# Group stmt by state
stmt = stmt.group_by(data.columns.name)

# Execute the statement and store all the records: results
results = conn.execute(stmt).fetchall()

# Print results
print(results)

# Print the keys/column names of the results returned
print(results[0].keys())


# In[ ]:


# Create a DataFrame from the results: df
df = pd.DataFrame(results)

# Set column names
df.columns = results[0].keys()

# Print the Dataframe
print(df)


# In[ ]:


plt.plot(df["aggregate salary"],"o")


# In[ ]:


df.plot.bar()
plt.show()


# In[ ]:


# # Import create_engine function
# from sqlalchemy import create_engine

# # Create an engine to the census database
# engine = create_engine('mysql+pymysql://student:datacamp@courses.csrrinzqubik.us-east-1.rds.amazonaws.com:3306/census')

# # Print the table names
# print(engine.table_names())


# In[ ]:


#!pip install pymysql


# In[ ]:


# Build query to return state names by population difference from 2008 to 2000: stmt
stmt = select([census.columns.state, (census.columns.pop2008-census.columns.pop2000).label('pop_change')])

# Append group by for the state: stmt_grouped
stmt_grouped = stmt.group_by(census.columns.state)

# Append order by for pop_change descendingly: stmt_ordered
stmt_ordered = stmt_grouped.order_by(desc('pop_change'))

# Return only 5 results: stmt_top5
stmt_top5 = stmt_ordered.limit(5)

# Use connection to execute stmt_top5 and fetch all results
results = connection.execute(stmt_top5).fetchall()

# Print the state and population change for each record
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))

# <script.py> output:
#     California:105705
#     Florida:100984
#     Texas:51901
#     New York:47098
#     Pennsylvania:42387


# In[ ]:


# import case, cast and Float from sqlalchemy
from sqlalchemy import case, cast, Float

# Build an expression to calculate female population in 2000
female_pop2000 = func.sum(
    case([
        (census.columns.sex == 'F', census.columns.pop2000)
    ], else_=0))

# Cast an expression to calculate total population in 2000 to Float
total_pop2000 = cast(func.sum(census.columns.pop2000), Float)

# Build a query to calculate the percentage of women in 2000: stmt
stmt = select([female_pop2000 / total_pop2000 * 100])

# Execute the query and store the scalar result: percent_female
percent_female = connection.execute(stmt).scalar()

# Print the percentage
print(percent_female)


# In[ ]:


# Build a statement to join census and state_fact tables: stmt
stmt = select([census.columns.pop2000, state_fact.columns.abbreviation])

# Execute the statement and get the first result: result
result = connection.execute(stmt).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))
#  pop2000 89600
#  abbreviation IL


# In[ ]:


# Build a statement to select the census and state_fact tables: stmt
stmt = select([census, state_fact])

# Add a select_from clause that wraps a join for the census and state_fact
# tables where the census state column and state_fact name column match
stmt_join = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name))

# Execute the statement and get the first result: result
result = connection.execute(stmt_join).first()

# Loop over the keys in the result object and print the key and value
for key in result.keys():
    print(key, getattr(result, key))

# state Illinois
#     sex M
#     age 0
#     pop2000 89600
#     pop2008 95012
#     id 13
#     name Illinois
#     abbreviation IL
#     country USA
#     type state
#     sort 10
#     status current
#     occupied occupied
#     notes 
#     fips_state 17
#     assoc_press Ill.
#     standard_federal_region V
#     census_region 2
#     census_region_name Midwest
#     census_division 3
#     census_division_name East North Central
#     circuit_court 7


# In[ ]:


# Build a statement to select the state, sum of 2008 population and census
# division name: stmt
stmt = select([
    census.columns.state,
    func.sum(census.columns.pop2008),
    state_fact.columns.census_division_name
])

# Append select_from to join the census and state_fact tables by the census state and state_fact name columns
stmt_joined = stmt.select_from(
    census.join(state_fact, census.columns.state == state_fact.columns.name)
)

# Append a group by for the state_fact name column
stmt_grouped = stmt_joined.group_by(state_fact.columns.name)

# Execute the statement and get the results: results
results = connection.execute(stmt_grouped).fetchall()

# Loop over the results object and print each record.
for record in results:
    print(record)


# In[ ]:


# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select names of managers and their employees: stmt
stmt = select(
    [managers.columns.name.label('manager'),
     employees.columns.name.label('employee')]
)

# Match managers id with employees mgr: stmt_matched
stmt_matched = stmt.where(managers.columns.id == employees.columns.mgr)

# Order the statement by the managers name: stmt_ordered
stmt_ordered = stmt_matched.order_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt_ordered).fetchall()

# Print records
for record in results:
    print(record)
# ('FILLMORE', 'GRANT')
#     ('FILLMORE', 'ADAMS')
#     ('FILLMORE', 'MONROE')
#     ('GARFIELD', 'JOHNSON')
#     ('GARFIELD', 'LINCOLN')
#     ('GARFIELD', 'POLK')
#     ('GARFIELD', 'WASHINGTON')
#     ('HARDING', 'TAFT')
#     ('HARDING', 'HOOVER')
#     ('JACKSON', 'HARDING')
#     ('JACKSON', 'GARFIELD')
#     ('JACKSON', 'FILLMORE')
#     ('JACKSON', 'ROOSEVELT')


# In[ ]:


# Make an alias of the employees table: managers
managers = employees.alias()

# Build a query to select names of managers and counts of their employees: stmt
stmt = select([managers.columns.name, func.count(employees.columns.id)])

# Append a where clause that ensures the manager id and employee mgr are equal: stmt_matched 
stmt_matched = stmt.where(managers.columns.id == employees.columns.mgr)

# Group by Managers Name: stmt_grouped
stmt_grouped = stmt_matched.group_by(managers.columns.name)

# Execute statement: results
results = connection.execute(stmt_grouped).fetchall()

# print manager
for record in results:
    print(record)
# ('FILLMORE', 3)
#     ('GARFIELD', 4)
#     ('HARDING', 2)
#     ('JACKSON', 4)

# In [1]: 


# In[ ]:


# Start a while loop checking for more results
while more_results:
    # Fetch the first 50 results from the ResultProxy: partial_results
    partial_results = results_proxy.fetchmany(50)

    # if empty list, set more_results to False
    if partial_results == []:
        more_results = False

    # Loop over the fetched records and increment the count for the state
    for row in partial_results:
        if row.state in state_count:
            state_count[row.state] += 1
        else:
            state_count[row.state] = 1

# Close the ResultProxy, and thus the connection
results_proxy.close()

# Print the count by state
print(state_count)


# In[ ]:


## Census case study
# Import create_engine, MetaData
from sqlalchemy import create_engine, MetaData

# Define an engine to connect to chapter5.sqlite: engine
engine = create_engine('sqlite:///chapter5.sqlite')

# Initialize MetaData: metadata
metadata = MetaData()


# Import Table, Column, String, and Integer
from sqlalchemy import Table, Column, String, Integer

# Build a census table: census
census = Table('census', metadata,
               Column('state', String(30)),
               Column('sex', String(1)),
               Column('age', Integer()),
               Column('pop2000', Integer()),
               Column('pop2008', Integer()))

# Create the table in the database
metadata.create_all(engine)

# Create an empty list: values_list
values_list = []

# Iterate over the rows
for row in csv_reader:
    # Create a dictionary with the values
    data = {'state': row[0], 'sex': row[1], 'age': row[2], 'pop2000': row[3],
            'pop2008': row[4]}
    # Append the dictionary to the values list
    values_list.append(data)

# Import insert
from sqlalchemy import insert

# Build insert statement: stmt
stmt = insert(census)

# Use values_list to insert data: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)



# Import select and func
from sqlalchemy import select, func

# Select the average of age weighted by pop2000
stmt = select([func.sum(census.columns.pop2000 * census.columns.age) 
 / func.sum(census.columns.pop2000)
 ])

# Import select and func
from sqlalchemy import select, func

# Relabel the new column as average_age
stmt = select([(func.sum(census.columns.pop2000 * census.columns.age) 
/ func.sum(census.columns.pop2000)).label('average_age')
 ])

# Import select and func
from sqlalchemy import select, func

# Add the sex column to the select statement
stmt = select([census.columns.sex,
    (func.sum(census.columns.pop2000 * census.columns.age) 
 / func.sum(census.columns.pop2000)).label('average_age')               
 ])

# Group by sex
stmt = stmt.group_by(census.columns.sex)


# Import select and func
from sqlalchemy import select, func

# Select sex and average age weighted by 2000 population
stmt = select([(func.sum(census.columns.pop2000 * census.columns.age) 
/ func.sum(census.columns.pop2000)).label('average_age'),
               census.columns.sex
])

# Group by sex
stmt = stmt.group_by(census.columns.sex)

# Execute the query and fetch all the results
results = connection.execute(stmt).fetchall()

# Print the sex and average age column for each result
for result in results:
    print(result.sex, result.average_age)
    
    
    
# import case, cast and Float from sqlalchemy
from sqlalchemy import case, cast, Float

# Build a query to calculate the percentage of women in 2000: stmt
stmt = select([census.columns.state,
    (func.sum(
        case([
            (census.columns.sex == 'F', census.columns.pop2000)
        ], else_=0)) /
     cast(func.sum(census.columns.pop2000), Float) * 100).label('percent_female')
])

# Group By state
stmt = stmt.group_by(census.columns.state)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the percentage
for result in results:
    print(result.state, result.percent_female)

    
# Build query to return state name and population difference from 2008 to 2000
stmt = select([census.columns.state,
     (census.columns.pop2008-census.columns.pop2000).label('pop_change')
])

# Group by State
stmt = stmt.group_by(census.columns.state)

# Order by Population Change
stmt = stmt.order_by(desc('pop_change'))

# Limit to top 10
stmt = stmt.limit(10)

# Use connection to execute the statement and fetch all results
results = connection.execute(stmt).fetchall()

# Print the state and population change for each record
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))
    
    
# California:105705
#     Florida:100984
#     Texas:51901
#     New York:47098
#     Pennsylvania:42387
#     Arizona:29509
#     Ohio:29392
#     Illinois:26221
#     Michigan:25126
#     North Carolina:24108
        
        
    
