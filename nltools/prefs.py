'''
NeuroLearn Preferences
======================


'''
__all__ = ['MNI_template','resolve_mni_path']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
from nltools.utils import get_resource_path
import six

MNI_template = dict(
    resolution = '2mm',
    mask = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'),
    plot =
    os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'),
    brain =
    os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz'),
)

def resolve_mni_path(MNI_template):
    """ Helper function to resolve MNI path based on dictionary setting."""

    res = MNI_template['resolution']
    if isinstance(res,six.string_types):
        if res == '3mm':
            MNI_template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain_mask.nii.gz')
            MNI_template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz')
            MNI_template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain.nii.gz')
        elif res == '2mm':
            MNI_template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz')
            MNI_template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
            MNI_template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
        else:
            raise ValueError("Available templates are '2mm' or '3mm'")
        return MNI_template
    else:
        raise TypeError("Available templates are '2mm' or '3mm'")

# class Prefs(object):
#
#     """
#     Prefs is a class to represent module level preferences for nltools, e.g. masks.
#     """
#
#     def __init__(self):
#         self.MNI_template = {}
#         self.MNI_template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz')
#         self.MNI_template['plot']=        os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
#         self.MNI_template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
#
#     def __repr__(self):
#         strOut = "nltools preferences:\n"
#         for section_name in ['MNI_template']:
#             section = getattr(self,section_name)
#             for key, val in list(section.items()):
#                 strOut += "%s['%s'] = %s\n" % (section_name, key, repr(val))
#         return strOut
#
#     def use_template(self,template_name):
#         if isinstance(template_name,six.string_types):
#             if template_name == '3mm':
#                 self.MNI_template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain_mask.nii.gz')
#                 self.MNI_template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz')
#                 self.MNI_template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain.nii.gz')
#             elif template_name == '2mm':
#                 self.MNI_template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz')
#                 self.MNI_template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
#                 self.MNI_template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
#             else:
#                 raise ValueError("Available templates are '2mm' or '3mm'")
#         else:
#             raise TypeError("Available templates are '2mm' or '3mm'")
