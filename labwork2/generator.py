import os
import random
import csv


def main():
    
     print("\n:: Linear Regression Dataset Generator ::\n\n")
    
     n = int(input("How many points : "))
     
     x = str(input("\nEnter X column name : "))
     y = str(input("\nEnter Y column name : "))
     
     coeff = random.randrange(0, 2)

     intercept = random.randrange(0, 1)
     
     path = str("./")
     
     myfile = "dataset_sam.csv"
     infilename = os.path.join(path,myfile)

     
     with open(infilename, 'w+') as csvfile:
         spamwriter = csv.writer(csvfile)
        

         for i in range(0,n+1):
         
             if (i == 0): 
                 list = [x , y]
                 spamwriter.writerow(list)
                 continue
             x = random.randrange(0, 5)
    
             y = coeff*x + intercept
    
             shuffle = random.randrange(0, 5)
    
             y += shuffle
             
             list = [x , y]
         
             spamwriter.writerow(list)
         
     print("\nFile created - dataset_sam.csv")
     


if __name__== "__main__":
  main()