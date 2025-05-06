data_string="""
ABSD 9:00 ERRE CDCD 
ABSD 9:00 ERRE DFDF FAIL
ABSD 9:00 ERRE DDFD PASS
ABSD 9:00 ERRE DFFF Pass
ABSD 9:00 ERRE FDFD COF
ABSD 9:00 ERRE IEOR FAIL  Failed operation
"""

def process_orders(data_string):
    #split by row
    new_data=data_string.split("\n")
    #print(new_data[1])

    for i  in range(len(new_data)):
        data=new_data[i]
        if i==0: #line for initial row 
            continue
        if data=="":
            continue
        print(data)
        data=data.split(" ")
        #print(data)
        if len(data)<5:
            print("missing data")
        if (len(data)>4):  
            if (data[4]!="PASS") and (data[4]!= "FAIL"):
                print("Incorrect input for  Pass or Fail Data")
        if  "PASS" not in data and "FAIL" not in data:
            print("Missing input for pass and fail")
        if (len(data)>5):
            print("too many inputs in row")
    return
process_orders(data_string)
