import pickle as p


def _using_file_sample():
    s = '''
    Programming is fun
    When ths work is done
    if you wanna make your work also fun:
        Use Python!
    '''

    f = open('tmp.txt', 'w')
    f.write(s)
    f.close()

    f = open('tmp.txt', 'r')
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        print(line)

    f.close()


def serialize():
    shoplistfile = 'shoplistfile.data'
    shoplist = ['apple', 'mango', 'banana', 'carrot']
    f = open(shoplistfile, 'wb')

    p.dump(shoplist, f)
    f.close()

    del shoplist

    # Read back from the storage
    f = open(shoplistfile, 'rb')
    shoplist = p.load(f)
    f.close()
    print(shoplist)


if __name__ == "__main__":
    serialize()
