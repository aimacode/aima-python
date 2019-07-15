
class Object:
    '''
    This represents any physical object that can appear in an Environment. You subclass Object to get the objects you
    want.  Each object can have a  .__name__  slot (used for output only).'''

   # Mark: __repr__ exists to create a printable output of an object (in this case, name)
    def __repr__(self):
        if self.id == '':
            return '<%s>' % getattr(self, '__name__', self.__class__.__name__)
        else:
            return '<%s id=%s>' % (getattr(self, '__name__', self.__class__.__name__), self.id)

    def is_alive(self):
        '''Objects that are 'alive' should return true.'''
        return hasattr(self, 'alive') and self.alive

    # is_grabbable()
    def is_grabbable(self, obj):
        return False

    def destroy(self):
        #print("Destroying %s" % self)
        pass

    # can the object be passed over, or does it occupy space.
    blocker = False
    #image_source = ''
    #image = None
    id = ''

class Dirt(Object):
    def __init__(self):
        pass

    def is_grabbable(self, obj):
        if hasattr(obj, 'holding'):
            return True
        else:
            return False

class Wall(Object):
    blocker = True

class DeadCell(Wall):
    pass

class Fire(Wall):
    def __repr__(self):
        if self.id == '':
            return '<%s t=%s>' % (getattr(self, '__name__', self.__class__.__name__), self.t)
        else:
            return '<%s t=%s id=%s>' % (getattr(self, '__name__', self.__class__.__name__), self.t, self.id)

    def __init__(self):
        self.t = 5