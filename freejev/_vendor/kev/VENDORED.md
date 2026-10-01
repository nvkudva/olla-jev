Vendored from https://github.com/jaredpalmer/kev at commit 90512f1c517d977741f2104470a40635408236c9
(Apache-2.0, see LICENSE): api.py, model.py, checkpoint.py, device.py.

Local change: checkpoint.read_meta loads head.pt with torch.load(weights_only=True), so a
tampered head.pt cannot run code.
