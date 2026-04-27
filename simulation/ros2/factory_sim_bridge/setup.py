from setuptools import setup

package_name = "factory_sim_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yyz",
    maintainer_email="yyz@example.com",
    description="Simulation-side ROS2 bridge for Factory MARL.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mock_sim_publisher = factory_sim_bridge.mock_sim_publisher:main",
        ],
    },
)
