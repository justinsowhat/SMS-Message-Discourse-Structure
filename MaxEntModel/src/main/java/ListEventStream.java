

import opennlp.model.Event;
import opennlp.model.EventStream;

import java.util.List;

/**
 * Created by justinso on 6/17/16.
 */
public class ListEventStream implements EventStream{

    List<Event> events;
    int currentIndex = 0;
    int numEvents;

    public ListEventStream(List<Event> events) {
        this.events = events;
        numEvents = events.size();
    }

    public Event next() {
        return events.get(currentIndex++);
    }

    public boolean hasNext() {
        return currentIndex < numEvents;
    }

}
