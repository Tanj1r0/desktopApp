<?php

namespace App\Http\Controllers;

use App\Models\Note;
use Illuminate\Http\Request;

class NoteController extends Controller
{
    public function index()
    {
        $notes = Note::all();
        return view('notes.index', compact('notes'));
    }

    public function create()
    {
        return view('notes.create');
    }

    public function store(Request $request)
    {
        $request->validate([
            'title' => 'required',
            'content' => 'nullable',
        ]);

        Note::create($request->all());

        return redirect('/')->with('success', 'Заметка создана');
    }

    public function destroy(Note $note)
    {
        $note->delete();
        return redirect('/')->with('success', 'Заметка удалена');
    }

    public function edit($id)
    {
        $note = \App\Models\Note::findOrFail($id);
        return view('notes.edit', compact('note'));
    }

    public function update(Request $request, Note $note)
    {
        $validated = $request->validate([
            'title' => 'required|string|max:255',
            'content' => 'required|string',
        ]);

        $note->update($validated);

        return redirect()->route('notes.index')->with('success', 'Заметка обновлена!');
    }
}
