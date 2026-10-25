"""Framework-free inference used by the deployments.

Modules here must not import ``qhcyber.modeling``: it imports torch at module level, and the
deployment images do not install torch.
"""
