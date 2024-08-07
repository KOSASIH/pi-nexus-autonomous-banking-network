import xml.etree.ElementTree as ET

class XCMPAdapter:
    def __init__(self, namespace: str):
        self.namespace = namespace

    def convert_to_xcmp(self, data: dict) -> ET.Element:
        root = ET.Element("XCMP", xmlns=self.namespace)
        for key, value in data.items():
            element = ET.SubElement(root, key)
            element.text = str(value)
        return root

    def convert_from_xcmp(self, xcmp: ET.Element) -> dict:
        data = {}
        for element in xcmp.iter():
            if element.text:
                data[element.tag] = element.text
        return data
