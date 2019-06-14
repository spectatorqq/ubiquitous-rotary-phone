import socket
import cv2
import numpy
from io import BytesIO
from PIL import Image
import struct

host = socket.gethostname()
port = 9393

address = (host, port)
headsize = 12

def byte_to_img(byte):
    str_buf = BytesIO(byte)
    img = numpy.array(Image.open(str_buf))
    str_buf.close()
    return img

def receivee2(process=None):
    buff = bytes()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(address)
    sock.listen(5)
    package, addr = sock.accept()
    while True:
        img = package.recv(1024)
        if img:
            print(type(img))
            buff = buff+img
            while True:
                if len(buff) < headsize:
                    print()
                    break
                headPack = struct.unpack('!3I', buff[:headsize])
                cmd = headPack[2]
                bodySize = headPack[1]
                if len(buff) < headsize+bodySize:
                    break
                body = buff[headsize:headsize+bodySize]
                if cmd == 101:
                    img = byte_to_img(body)
                    cv2.imshow("receive", img)

                print(len(body))
                buff = buff[headsize+bodySize:]
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                continue

        break
    cv2.destroyAllWindows()
    sock.close()


if __name__ == '__main__':
    receivee2()

