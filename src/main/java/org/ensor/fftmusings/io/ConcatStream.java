/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package org.ensor.fftmusings.io;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author jona
 */
public class ConcatStream extends InputStream
{
    private final List<InputStream> mStreams;
    private long mSize;

    public ConcatStream() {
        mStreams = new ArrayList<>();
        mSize = 0;
    }
    public void addStream(InputStream s, long size) {
        mStreams.add(s);
        mSize += size;
    }

    public long size() {
        return mSize;
    }
    
    public int read() throws IOException
    {
        if (mStreams.isEmpty()) {
            return -1;
        }
        InputStream is = mStreams.get(0);
        int c = is.read();
        if (c == -1) {
            mStreams.remove(0);
            return read();
        }
        return c;
    }
}
