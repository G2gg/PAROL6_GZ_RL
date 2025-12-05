from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'parol6_drawing'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add folders with all their contents recursively
        (os.path.join('share', package_name, 'svg_files'), glob('svg_files/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gunesh_pop_nvidia',
    maintainer_email='gunesh_pop_nvidia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_draw = parol6_drawing.robot_draw:main',
            'robot_draw_using_gui = parol6_drawing.robot_draw_using_gui:main',
            'web_gui = parol6_drawing.web_gui:main',
            'path_tracker = parol6_drawing.path_tracker:main'
        ],
    },
)
