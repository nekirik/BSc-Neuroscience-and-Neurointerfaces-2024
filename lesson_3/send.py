# send.py
import socket, sys
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.sendto((sys.argv[1] + "\n").encode(), ("172.20.10.2", 9999))