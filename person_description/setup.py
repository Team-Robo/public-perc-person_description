from setuptools import find_packages, setup

package_name = 'person_description'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nanyang Technological University Team Robo',
    maintainer_email='teamrobontu@gmail.com',
    description='Visual appearance description for a tracked person, via VLM',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'person_description = person_description.person_description:main',
        ],
    },
)
