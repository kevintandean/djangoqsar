from django.conf.urls import include, url
from django.contrib import admin
from django.conf.urls.static import static
from qsarmodelling import settings

urlpatterns = [
    # Examples:
    # url(r'^$', 'qsarmodelling.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^getdescriptor','qsar.views.get_descriptor', name='get_descriptor'),
    url(r'^test','qsar.views.score_test', name='test'),
    url(r'^getsmiles/(?P<name>\w+)','qsar.views.get_smiles',name='get_smiles'),
    url(r'^runfda','qsar.views.run_fda',name='run_fda'),
    url(r'^result','qsar.views.get_result', name='get_result'),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
