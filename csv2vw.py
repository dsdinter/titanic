'convert from [titanic-specific] CSV to VW format'

import sys, csv


input_file = sys.argv[1]
output_file = sys.argv[2]

reader = csv.reader( open( input_file ))
o = open( output_file, 'wb' )

counter = 0  
headers = reader.next()  
for line in reader:

    counter += 1
    if input_file == 'Data/train.csv':
        passenger_id = line[0]
        Survived = line[1]
        Pclass = line[2]
        Name = line[3]
        Sex = line[4]
        if Sex == 'male':
            Sexcod = '0'
        else:
            Sexcod = '1'
        Age = line[5]
        SibSp = line[6]
        Parch = line[7]
        Ticket = line[8]
        Fare = line[9]
        Cabin = line[10]
        Embarked = line[11]
        if Survived == '0':
            label = '-1'
        else:
            label = '1'
    else:
        passenger_id = line[0]
        Pclass = line[1]
        Name = line[2]
        Sex = line[3]
        if Sex == 'male':
            Sexcod = '0'
        else:
            Sexcod = '1'
        Age = line[4]
        SibSp = line[5]
        Parch = line[6]
        Ticket = line[7]
        Fare = line[8]
        Cabin = line[9]
        Embarked = line[10]
        label = '-1'
        
    output_line = "%s %s %s" % ( label, 1, passenger_id )     # weight is 1
    output_line += "|f %s %s %s %s %s %s" % ( Pclass, Sexcod, Age, SibSp, Parch, Fare)
    output_line += "\n"

    o.write( output_line )

    if counter % 100000 == 0:
        print counter