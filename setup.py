from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, 'pure_pursuit', 'follow_the_gap'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/pure_pursuit/config', glob('pure_pursuit/config/*.yaml')),
        ('share/' + package_name + '/pure_pursuit/launch', glob('pure_pursuit/launch/*.py')),
        ('share/' + package_name + '/pure_pursuit/racelines', glob('pure_pursuit/racelines/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel Baadom',
    maintainer_email='danny@mtsu.edu',
    description='F1Tenth autonomous racing project',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = pure_pursuit.pure_pursuit_node:main',
            'waypoint_visualizer_node = pure_pursuit.waypoint_visualizer_node:main',
            'gap_follow_node = follow_the_gap.reactive_node:main',
        ],
    },
)
