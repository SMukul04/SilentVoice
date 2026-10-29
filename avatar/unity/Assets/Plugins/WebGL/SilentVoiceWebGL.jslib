mergeInto(LibraryManager.library, {
    SendAvatarEvent: function(eventJsonPtr) {
        if (typeof window.onAvatarEvent === 'function') {
            var eventJson = UTF8ToString(eventJsonPtr);
            window.onAvatarEvent(eventJson);
        }
    }
});
