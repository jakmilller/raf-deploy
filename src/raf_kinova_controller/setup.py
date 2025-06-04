from setuptools import setup
import os
from glob import glob

package_name = 'raf_kinova_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Add other data files here if needed (e.g., config files)
    ],
    install_requires=['setuptools', 'rclpy', 'tf_transformations', 'numpy'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='ROS2 controller for Kinova Gen3 arm using Kortex Python API',
    license='Apache License 2.0', # Or your preferred license
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinova_controller = raf_kinova_controller.kinova_controller_node:main',
        ],
    },
)