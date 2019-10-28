import mlxtend as ml
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
import pandas as pd
import csv
import timeit


def main():
    data = {}
    with open('online_retail.csv') as f:
        header_row = False
        header = []
        for cols in csv.reader(f):
            if not header_row:
                header_row = True
                header = [i for i in cols]  # list of headers
                for h in header:
                    data[h] = []  # init of dictionary
                continue
            if len(cols) == len(header) and cols[0][0] != 'C' and cols[0] != '' and cols[
                2] != '':  # chech for empty InvoiceIds or Descriptions
                for col, name in zip(cols, header):
                    data[name].append(col)

    byinvoice = {}  # dictionary with data grouped by invoice
    for invoiceNo, desc in zip(data['InvoiceNo'], data['Description']):
        if invoiceNo not in byinvoice.keys():
            byinvoice[invoiceNo] = []  # if the dict doesn't have an invoice entry, initialize it

        byinvoice[invoiceNo].append(desc)

    unique_products = []  # set of unique products, is used for the bitmap matrix
    for prod in data['Description']:
        if prod not in unique_products:
            unique_products.append(prod)

    match_matrix = []  # bitmap matrix is done by a list of lists

    for inv in byinvoice.keys():
        match_array = [0] * len(unique_products)  # init of each row of bitmap matrix
        for prod in byinvoice[
            inv]:  # for every product of a given invoice, I get the index inside the list of unique products
            i = unique_products.index(prod)
            match_array[i] = 1  # that index is used to put 1 in the correct position inside current matrix row

        match_matrix.append(match_array)

    df = pd.DataFrame(data=match_matrix, columns=unique_products)

    '''
    fi = fpgrowth(df, 0.05)
    print(len(fi))
    print(fi.to_string())
    
    # checking if result is right (first product is present in 10.9 % of invoices)
    i = 0
    for inv in byinvoice.values():
        if unique_products[0] in inv:
            i += 1
    print(str(i/len(byinvoice.keys())))
    '''

    # time check
    # I did it with 0.05 because using minsup=0.01 with apriori resulted in a Memory Error :(((((
    print(timeit.timeit(lambda: apriori(df, 0.05), number=1))
    print(timeit.timeit(lambda: fpgrowth(df, 0.05), number=1))

    fi = fpgrowth(df, 0.01)
    print(len(fi))
    fi_list = fi.values.tolist()

    # just to see the top 10 relevant informations
    # the larger the set, the more relevant the information     (I think)
    fi_list.sort(key=lambda x: -len(x[1]))
    print(fi_list[:10])  # top 10

    ar = association_rules(fi, metric="confidence", min_threshold=0.85)
    ar.to_csv(r'association_rules.csv', header=True)  # dump to file for visualization reasons


if __name__ == '__main__':
    main()
