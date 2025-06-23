from setuptools import find_packages, setup

package_name = 'py_node_tutorial'

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
    maintainer='kimsooyoung',
    maintainer_email='tge1375@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example_node_1 = py_node_tutorial.node_example_1:main',
            'example_node_2 = py_node_tutorial.node_example_2:main',
            'example_node_3 = py_node_tutorial.node_example_3:main',
        ],
    },
)
