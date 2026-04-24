from setuptools import setup

package_name = "factory_core"

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
    description="Core control layer for Factory MARL ROS2 integration.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "core_control_node = factory_core.control_node:main",
            "core_visualization_node = factory_core.visualization_node:main",
        ],
    },
)
