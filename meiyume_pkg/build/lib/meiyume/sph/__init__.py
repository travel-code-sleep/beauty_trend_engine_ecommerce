# import sph
# import sph.crawler
# import sph.master
# import sph.cleaner
# import sph.app
try:
    __import__('pkg_resources').declare_namespace(__name__)
except ImportError:
    __path__ = __import__('pkgutil').extend_path(__path__, __name__)