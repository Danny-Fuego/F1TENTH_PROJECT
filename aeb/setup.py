from setuptools import find_packages, setup

package_name = 'aeb'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Daniel Baadom',
    maintainer_email='danielbaadom7@gmail.com',
    description='Automatic emergency braking (time to collision) for F1Tenth',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aeb_node = aeb.aeb_node:main',
        ],
    },
)
