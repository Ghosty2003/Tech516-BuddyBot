from setuptools import find_packages, setup

package_name = 'yolo_detect'

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
           'yolo = yolo_detect.yolo:main',
           'follow = yolo_detect.follow:main', 
           'follow_face = yolo_detect.follow_face:main', 
           'manual = yolo_detect.manual_action_input:main',
           'platform = yolo_detect.platform_height_controller:main',
           'gesture = yolo_detect.gesture:main'
        ],
    },
)
