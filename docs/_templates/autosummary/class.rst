{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members: Module, object

   {# `BAE` subclasses torch.nn.Module, whose 65 public methods (add_module,
      cuda, xpu, register_load_state_dict_pre_hook, ...) and attributes
      (T_destination, call_super_init, dump_patches) otherwise bury the 18 that
      belong to this package. `:inherited-members:` above takes the base classes
      to *stop* at, so members defined on Module and object are excluded while
      genuine inheritance elsewhere still shows. #}

   {% block methods %}
   {% set own_methods = methods | reject("in", inherited_members) | list %}
   {% if own_methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in own_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set own_attributes = attributes | reject("in", inherited_members) | list %}
   {% if own_attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in own_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
