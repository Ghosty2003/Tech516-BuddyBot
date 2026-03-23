from setuptools import find_packages, setup
import os

package_name = 'buddybot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/launch.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
           'yolo = buddybot.yolo:main',
           'follow = buddybot.follow:main', 
           'follow_face = buddybot.follow_face:main', 
           'manual = buddybot.manual_action_input:main',
           'platform = buddybot.platform_height_controller:main',
           'gesture = buddybot.gesture:main'
        ],
    },
)
