import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def process(font, size, chars):
    proc = subprocess.Popen([
        './load_font',
        font,
        size,
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    for c in chars:
        charbuf = c.encode("utf-32-le")
        proc.stdin.write(charbuf[:4])
        proc.stdin.flush()
        result = proc.stdout.read(32)

        code = result[:4]
        rows = int.from_bytes(result[4:8], 'little')
        width = int.from_bytes(result[8:12], 'little')
        boundingWidth = int.from_bytes(result[12:16], 'little', signed=True)
        boundingHeight = int.from_bytes(result[16:20], 'little', signed=True)
        horiBearingX = int.from_bytes(result[20:24], 'little', signed=True)
        horiBearingY = int.from_bytes(result[24:28], 'little', signed=True)
        horiAdvance = int.from_bytes(result[28:32], 'little', signed=True)

        if rows == 0 or width == 0:
            continue

        boundingWidth = boundingWidth / 64
        boundingHeight = boundingHeight / 64
        horiBearingX = horiBearingX / 64
        horiBearingY = horiBearingY / 64
        horiAdvance = horiAdvance / 64

        print('code', code.decode("utf-32-le"),
            'rows:', rows,
            'width:', width,
            'boundingWidth:', boundingWidth,
            'boundingHeight:', boundingHeight,
            'horiBearingX:', horiBearingX,
            'horiBearingY:', horiBearingY,
            'horiAdvance:', horiAdvance)

        buffer = proc.stdout.read(rows*width)
        img = np.frombuffer(buffer, dtype='ubyte').reshape(rows,width)

        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray', extent=[horiBearingX, horiBearingX + boundingWidth, horiBearingY - boundingHeight, horiBearingY])
        points = [
            [0,0],
            [horiAdvance, 0],
            [horiAdvance, int(size)],
            [0, int(size)],
            [0,0],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1], color="g")
        plt.gca().set_aspect('equal')

        result = proc.stdout.read(28)

        rows = int.from_bytes(result[:4], 'little')
        width = int.from_bytes(result[4:8], 'little')
        boundingWidth = int.from_bytes(result[8:12], 'little', signed=True)
        boundingHeight = int.from_bytes(result[12:16], 'little', signed=True)
        vertBearingX = int.from_bytes(result[16:20], 'little', signed=True)
        vertBearingY = int.from_bytes(result[20:24], 'little', signed=True)
        vertAdvance = int.from_bytes(result[24:28], 'little', signed=True)

        boundingWidth = boundingWidth / 64
        boundingHeight = boundingHeight / 64
        vertBearingX = vertBearingX / 64
        vertBearingY = vertBearingY / 64
        vertAdvance = vertAdvance / 64

        print('code', code.decode("utf-32-le"),
            'rows:', rows,
            'width:', width,
            'boundingWidth:', boundingWidth,
            'boundingHeight:', boundingHeight,
            'vertBearingX:', vertBearingX,
            'vertBearingY:', vertBearingY,
            'vertAdvance:', vertAdvance)

        buffer = proc.stdout.read(rows*width)
        img = np.frombuffer(buffer, dtype='ubyte').reshape(rows,width)

        plt.subplot(1,2,2)
        plt.imshow(img, cmap='gray', extent=[vertBearingX, vertBearingX + boundingWidth, -(vertBearingY+boundingHeight), -vertBearingY])
        points = [
            [0,0],
            [0, -vertAdvance],
            [-int(size)/2, -vertAdvance],
            [-int(size)/2, 0],
            [int(size)/2, 0],
            [int(size)/2, -vertAdvance],
            [0, -vertAdvance],
        ]
        points = np.array(points)
        plt.plot(points[:,0], points[:,1], color="g")
        plt.gca().set_aspect('equal')

        plt.show()

if __name__ == '__main__':
    process(*sys.argv[1:])
