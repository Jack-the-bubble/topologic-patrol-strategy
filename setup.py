from setuptools import setup

setup(
   name='leader_follower',
   version='0.0.1',
   description='First version',
   author='me',
   author_email='foomail@foo.com',
   packages=['leader_follower'],  # would be the same as name
   install_requires=['scipy'], #external packages acting as dependencies
)