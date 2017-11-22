'''
NeuroLearn Preferences
======================


'''
__all__ = ['MNI_Template','resolve_mni_path']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import os
from nltools.utils import get_resource_path
import six

MNI_Template = dict(
    resolution = '2mm',
    mask_type = 'with_ventricles',
    mask = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz'),
    plot =
    os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz'),
    brain =
    os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz'),
)

def resolve_mni_path(MNI_Template):
    """ Helper function to resolve MNI path based on MNI_Template prefs setting."""

    res = MNI_Template['resolution']
    m = MNI_Template['mask_type']
    assert isinstance(res,six.string_types), "tesolution must be provided as  a string!"
    assert isinstance(m,six.string_types), "mask_type must be provided as a string!"

    if res == '3mm':
        if m == 'with_ventricles':
            MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain_mask.nii.gz')
        elif m == 'no_ventricles':
            MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain_mask_no_ventricles.nii.gz')
        else:
            raise ValueError("Available mask_types are 'with_ventricles' or 'no_ventricles'")

        MNI_Template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz')

        MNI_Template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain.nii.gz')

    elif res == '2mm':
        if m == 'with_ventricles':
            MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz')
        elif m == 'no_ventricles':
            MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask_no_ventricles.nii.gz')
        else:
            raise ValueError("Available mask_types are 'with_ventricles' or 'no_ventricles'")

        MNI_Template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')

        MNI_Template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
    else:
        raise ValueError("Available templates are '2mm' or '3mm'")
    return MNI_Template

# class Prefs(object):
#
#     """
#     Prefs is a class to represent module level preferences for nltools, e.g. masks.
#     """
#
#     def __init__(self):
#         self.MNI_Template = {}
#         self.MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz')
#         self.MNI_Template['plot']=        os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
#         self.MNI_Template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
#
#     def __repr__(self):
#         strOut = "nltools preferences:\n"
#         for section_name in ['MNI_Template']:
#             section = getattr(self,section_name)
#             for key, val in list(section.items()):
#                 strOut += "%s['%s'] = %s\n" % (section_name, key, repr(val))
#         return strOut
#
#     def use_template(self,template_name):
#         if isinstance(template_name,six.string_types):
#             if template_name == '3mm':
#                 self.MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain_mask.nii.gz')
#                 self.MNI_Template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_3mm.nii.gz')
#                 self.MNI_Template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_3mm_brain.nii.gz')
#             elif template_name == '2mm':
#                 self.MNI_Template['mask'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask.nii.gz')
#                 self.MNI_Template['plot'] =        os.path.join(get_resource_path(),'MNI152_T1_2mm.nii.gz')
#                 self.MNI_Template['brain'] = os.path.join(get_resource_path(),'MNI152_T1_2mm_brain.nii.gz')
#             else:
#                 raise ValueError("Available templates are '2mm' or '3mm'")
#         else:
#             raise TypeError("Available templates are '2mm' or '3mm'")
