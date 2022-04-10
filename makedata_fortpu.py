from dataset import FontData, make_tfrecodes

gspath='gs://your-bucket-name/'

def make1():
    d = FontData()
    make_tfrecodes(d, gspath, 512, 64, 8, train=True)
    make_tfrecodes(d, gspath, 512, 64, 8, train=False)

if __name__ == "__main__":
    make1()
