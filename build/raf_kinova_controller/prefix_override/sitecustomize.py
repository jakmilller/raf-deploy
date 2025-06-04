import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mcrr-lab/raf-deploy/install/raf_kinova_controller'
