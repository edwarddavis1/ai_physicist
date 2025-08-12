"""
LaTeX rendering utilities for physics problems
"""
import re
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class LaTeXRenderer:
    """Simple LaTeX renderer for terminal display"""
    
    def __init__(self):
        self.console = Console()
    
    def simplify_latex(self, text):
        """Convert LaTeX to plain text with Unicode symbols"""
        # Replace common LaTeX symbols with Unicode
        replacements = {
            r'\\hbar': 'ℏ',
            r'\\omega': 'ω',
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\Delta': 'Δ',
            r'\\epsilon': 'ε',
            r'\\theta': 'θ',
            r'\\lambda': 'λ',
            r'\\mu': 'μ',
            r'\\nu': 'ν',
            r'\\pi': 'π',
            r'\\rho': 'ρ',
            r'\\sigma': 'σ',
            r'\\tau': 'τ',
            r'\\phi': 'φ',
            r'\\chi': 'χ',
            r'\\psi': 'ψ',
            r'\\Psi': 'Ψ',
            r'\\sum': 'Σ',
            r'\\int': '∫',
            r'\\partial': '∂',
            r'\\nabla': '∇',
            r'\\infty': '∞',
            r'\\cdot': '·',
            r'\\times': '×',
            r'\\pm': '±',
            r'\\leq': '≤',
            r'\\geq': '≥',
            r'\\neq': '≠',
            r'\\approx': '≈',
            r'\\equiv': '≡',
            r'\\sqrt': '√',
            r'\\frac': '',
            r'\\left': '',
            r'\\right': '',
        }
        
        result = text
        for latex, unicode_char in replacements.items():
            result = re.sub(latex, unicode_char, result)
        
        # Remove LaTeX commands
        result = re.sub(r'\\[a-zA-Z]+\*?', '', result)
        
        # Clean up brackets and braces
        result = re.sub(r'\{([^}]*)\}', r'\1', result)
        result = re.sub(r'\\\(([^)]*)\\\)', r'\1', result)
        result = re.sub(r'\\\[([^]]*)\\\]', r'\n    \1\n', result)
        
        return result
    
    def display_problem(self, question, answer, problem_idx=None):
        """Display a physics problem with simplified LaTeX"""
        title = f"Physics Problem {problem_idx}" if problem_idx is not None else "Physics Problem"
        
        # Get terminal width for responsive scaling
        terminal_width = self.console.size.width
        panel_width = min(terminal_width - 4, 120)  # Leave 4 chars margin, max 120 chars
        
        # Process question
        simplified_question = self.simplify_latex(question)
        question_panel = Panel(
            simplified_question,
            title="[bold blue]Question[/bold blue]",
            border_style="blue",
            width=panel_width
        )
        
        # Process answer
        simplified_answer = self.simplify_latex(answer)
        answer_panel = Panel(
            simplified_answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            width=panel_width
        )
        
        self.console.print(f"\n[bold magenta]{title}[/bold magenta]")
        self.console.print(question_panel)
        self.console.print(answer_panel)
        
        return simplified_question, simplified_answer
    
    def display_model_response(self, response, model_name=None):
        """Display a model response with simplified LaTeX"""
        title = f"Model Response ({model_name})" if model_name else "Model Response"
        
        # Get terminal width for responsive scaling
        terminal_width = self.console.size.width
        panel_width = min(terminal_width - 4, 120)  # Leave 4 chars margin, max 120 chars
        
        # Process response
        simplified_response = self.simplify_latex(response)
        response_panel = Panel(
            simplified_response,
            title="[bold yellow]Model Response[/bold yellow]",
            border_style="yellow",
            width=panel_width
        )
        
        self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
        self.console.print(response_panel)
        
        return simplified_response
    
    def display_problem_with_model_response(self, question, answer, model_response, problem_idx=None, model_name=None):
        """Display a physics problem with answer and model response"""
        title = f"Physics Problem {problem_idx}" if problem_idx is not None else "Physics Problem"
        model_title = f" ({model_name})" if model_name else ""
        
        # Get terminal width for responsive scaling
        terminal_width = self.console.size.width
        panel_width = min(terminal_width - 4, 120)  # Leave 4 chars margin, max 120 chars
        
        # Process question
        simplified_question = self.simplify_latex(question)
        question_panel = Panel(
            simplified_question,
            title="[bold blue]Question[/bold blue]",
            border_style="blue",
            width=panel_width
        )
        
        # Process correct answer
        simplified_answer = self.simplify_latex(answer)
        answer_panel = Panel(
            simplified_answer,
            title="[bold green]Correct Answer[/bold green]",
            border_style="green",
            width=panel_width
        )
        
        # Process model response
        simplified_response = self.simplify_latex(model_response)
        response_panel = Panel(
            simplified_response,
            title=f"[bold yellow]Model Response{model_title}[/bold yellow]",
            border_style="yellow",
            width=panel_width
        )
        
        self.console.print(f"\n[bold magenta]{title}[/bold magenta]")
        self.console.print(question_panel)
        self.console.print(answer_panel)
        self.console.print(response_panel)
        
        return simplified_question, simplified_answer, simplified_response


if __name__ == "__main__":
    from datasets import load_dataset
    
    # Load the data
    dataset = load_dataset("UGPhysics/ugphysics", "QuantumMechanics", split="en")
    
    # Create renderer
    renderer = LaTeXRenderer()
    
    # Display first problem
    question = dataset['problem'][0]
    answer = dataset['solution'][0]
    
    renderer.display_problem(question, answer, 0)
